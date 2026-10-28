//! Registered source storage shared by local, shared-storage and device reads.

use std::collections::HashMap;
use std::sync::Arc;

use super::Retirement;
use super::export_table::{Source, SourceKey};
use crate::{Error, Result};

#[derive(PartialEq)]
enum Status {
    Active,
    Revoked,
    Reclaiming,
}

/// One registered source and its physical retirement signal. A source stays
/// registered through producer completion, granted reads and reclamation.
pub(crate) struct RegisteredBuffer {
    source: Source,
    retirement: Arc<Retirement>,
    readers: usize,
    status: Status,
}

impl RegisteredBuffer {
    pub(crate) fn source(&self) -> &Source {
        &self.source
    }

    pub(crate) fn retirement(&self) -> &Arc<Retirement> {
        &self.retirement
    }

    pub(crate) fn acquire(&mut self) -> Result<&Source> {
        if self.status != Status::Active {
            return Err(Error::Invalid(
                "buffer is no longer available for reading".into(),
            ));
        }
        self.readers += 1;
        Ok(&self.source)
    }

    /// The grant remains held after revocation, until physical access ends.
    pub(crate) fn release_reader(&mut self) -> Result<()> {
        self.readers = self
            .readers
            .checked_sub(1)
            .ok_or_else(|| Error::Invariant("buffer read was already released".into()))?;
        Ok(())
    }

    pub(crate) fn release(&mut self) {
        if self.status == Status::Active {
            self.status = Status::Revoked;
        }
    }

    /// Check producer completion and remote acknowledgments before handing
    /// storage to the reclaimer.
    /// The query runs under the owner's lock so another reaper cannot unmap
    /// its backing. It must be read-only and must not invoke observers.
    pub(crate) fn begin_reclaim(
        &mut self,
        settled: impl FnOnce(&Source) -> Result<bool>,
    ) -> Result<Option<(&Source, &Arc<Retirement>)>> {
        if self.status != Status::Revoked || self.readers != 0 {
            return Ok(None);
        }
        if !settled(&self.source)? {
            return Ok(None);
        }

        self.status = Status::Reclaiming;
        Ok(Some((&self.source, &self.retirement)))
    }
}

/// Registration and retirement of transport source storage. The registry
/// counts nothing: every export reserves its bytes from the rank's transfer
/// capacity before it registers, which refuses what the rank cannot hold.
/// The transport serializes mutations and performs reclamation outside its
/// ownership lock. Backend source values carry no registry lifecycle state.
pub(crate) struct BufferRegistry {
    closing: bool,
    buffers: HashMap<SourceKey, RegisteredBuffer>,
}

impl BufferRegistry {
    pub(crate) fn new() -> Self {
        Self {
            closing: false,
            buffers: HashMap::new(),
        }
    }

    pub(crate) fn register(
        &mut self,
        key: SourceKey,
        source: Source,
        retirement: Arc<Retirement>,
    ) -> Result<()> {
        if self.closing {
            return Err(Error::Resource("the transport is closing"));
        }
        if self.buffers.contains_key(&key) {
            return Err(Error::Invalid("buffer is already registered".into()));
        }

        self.buffers.insert(
            key,
            RegisteredBuffer {
                source,
                retirement,
                readers: 0,
                status: Status::Active,
            },
        );
        Ok(())
    }

    pub(crate) fn get_mut(&mut self, key: &SourceKey) -> Option<&mut RegisteredBuffer> {
        self.buffers.get_mut(key)
    }

    pub(crate) fn buffers(&self) -> impl Iterator<Item = &RegisteredBuffer> {
        self.buffers.values()
    }

    pub(crate) fn revoked(&self) -> impl Iterator<Item = &SourceKey> {
        self.buffers
            .iter()
            .filter(|(_, buffer)| buffer.status == Status::Revoked)
            .map(|(key, _)| key)
    }

    pub(crate) fn awaiting_acknowledgment(&self) -> bool {
        self.buffers
            .values()
            .any(|buffer| buffer.status == Status::Revoked)
    }

    /// Return successfully retired buffers for destruction outside the owner
    /// lock. A failed or cancelled retirement still retains its backing.
    pub(crate) fn take_finished(&mut self) -> Vec<RegisteredBuffer> {
        self.buffers
            .extract_if(|_, buffer| {
                buffer.status == Status::Reclaiming && buffer.retirement.succeeded()
            })
            .map(|(_, buffer)| buffer)
            .collect()
    }

    /// Stop admission and revoke reads before the backend drains reclamation.
    pub(crate) fn close(&mut self) {
        self.closing = true;
        for buffer in self.buffers.values_mut() {
            buffer.release();
        }
    }

    pub(crate) fn require_retired(&self) -> Result<()> {
        if !self.buffers.is_empty() {
            return Err(Error::Resource(
                "buffer completion or readers remain unresolved; storage is retained",
            ));
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Weak;

    use super::*;
    use crate::ffi::{Backing, DType};
    use crate::storage::allocation::{Storage, StorageView};
    use crate::transfer::backing::TransportBuffer;
    use crate::transfer::{ByteReservation, Spans, TransferCapacity};

    /// A registered local export of four host bytes and a weak reference
    /// to the storage it retains.
    fn source(
        key: u64,
        capacity: &Arc<TransferCapacity>,
    ) -> Result<(SourceKey, Source, Weak<Storage>)> {
        let storage = Storage::host(4);
        let retained = Arc::downgrade(&storage);
        let spans = Spans::single(StorageView::contiguous(
            storage as Arc<dyn Backing>,
            0,
            DType::U8,
            &[4],
        )?)?;
        let buffer = TransportBuffer::local(spans, None, ByteReservation::new(capacity, 4)?);
        Ok((
            SourceKey::Local(key),
            Source::new(Arc::new(buffer)),
            retained,
        ))
    }

    fn capacity() -> Result<Arc<TransferCapacity>> {
        Ok(Arc::new(TransferCapacity::new(64, 1)?))
    }

    #[test]
    fn reclamation_waits_for_readers_and_settlement() -> Result<()> {
        let capacity = capacity()?;
        let mut registry = BufferRegistry::new();
        let (key, first, retained) = source(1, &capacity)?;
        let retirement = Arc::new(Retirement::default());
        registry.register(key.clone(), first, Arc::clone(&retirement))?;

        let buffer = registry
            .get_mut(&key)
            .ok_or(Error::State("registered source is missing"))?;
        buffer.acquire()?;
        buffer.release();
        assert!(
            buffer.acquire().is_err(),
            "a revoked source admits no reader"
        );
        assert!(
            buffer.begin_reclaim(|_| Ok(true))?.is_none(),
            "a reader holds it"
        );
        buffer.release_reader()?;
        assert!(
            buffer.begin_reclaim(|_| Ok(false))?.is_none(),
            "readers have not settled"
        );
        assert!(buffer.begin_reclaim(|_| Ok(true))?.is_some());
        assert!(registry.require_retired().is_err());

        // Successful physical retirement returns the storage.
        retirement.complete(Ok(()))?;
        drop(registry.take_finished());
        assert!(retained.upgrade().is_none());
        registry.require_retired()?;
        let (next, second, _) = source(2, &capacity)?;
        registry.register(next.clone(), second, Arc::new(Retirement::default()))?;

        // Closing revokes every source and admits no registration.
        registry.close();
        let buffer = registry
            .get_mut(&next)
            .ok_or(Error::State("replacement source missing"))?;
        assert!(buffer.acquire().is_err());
        let (last, third, _) = source(3, &capacity)?;
        assert!(
            registry
                .register(last, third, Arc::new(Retirement::default()))
                .is_err()
        );
        Ok(())
    }

    #[test]
    fn an_unsettled_source_retains_only_its_own_storage() -> Result<()> {
        let capacity = capacity()?;
        let mut registry = BufferRegistry::new();
        let (pending, first, retained) = source(1, &capacity)?;
        let (independent, second, _) = source(2, &capacity)?;
        registry.register(pending.clone(), first, Arc::new(Retirement::default()))?;
        registry.register(independent.clone(), second, Arc::new(Retirement::default()))?;

        let buffer = registry
            .get_mut(&pending)
            .ok_or(Error::State("pending source missing"))?;
        buffer.release();
        assert!(buffer.begin_reclaim(|_| Ok(false))?.is_none());

        let buffer = registry
            .get_mut(&independent)
            .ok_or(Error::State("independent source missing"))?;
        buffer.release();
        let (_, completion) = buffer.begin_reclaim(|_| Ok(true))?.ok_or(Error::State(
            "independent source did not become reclaimable",
        ))?;
        crate::transfer::notify(completion.complete(Ok(()))?);
        drop(registry.take_finished());

        let (next, third, _) = source(3, &capacity)?;
        registry.register(next, third, Arc::new(Retirement::default()))?;
        assert!(retained.upgrade().is_some());
        assert!(registry.require_retired().is_err());
        Ok(())
    }

    #[test]
    fn a_failed_reclamation_keeps_its_storage() -> Result<()> {
        let capacity = capacity()?;
        let mut registry = BufferRegistry::new();
        let (key, first, retained) = source(1, &capacity)?;
        let retirement = Arc::new(Retirement::default());
        registry.register(key.clone(), first, Arc::clone(&retirement))?;
        let buffer = registry
            .get_mut(&key)
            .ok_or(Error::State("source missing"))?;
        buffer.release();
        assert!(buffer.begin_reclaim(|_| Ok(true))?.is_some());

        retirement.complete(Err(Error::transport(
            "reclamation could not establish device completion",
        )))?;
        drop(registry.take_finished());
        assert!(registry.get_mut(&key).is_some());
        assert!(retained.upgrade().is_some());
        assert!(registry.require_retired().is_err());
        Ok(())
    }
}
