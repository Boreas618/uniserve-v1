//! The local mechanism: a process-local export registers the producer's
//! spans with its fence in the transport's export table; a reader in this
//! process borrows them in place or copies them into its destination.

use std::ops::Range;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use uniserve_ipc::{Locator, TransferTransport};

use super::backing::TransportBuffer;
use super::export_table::ExportTable;
use super::read::{self, NativeRead, Read, ReadCall};
use super::transport::{ExportView, Transport, producer, read_destination, within};
use super::{ByteReservation, ReadReservation, Spans};
use crate::ffi::MemoryKind;
use crate::stream::Issue;
use crate::{Error, Result, lock};

impl Transport {
    pub(super) fn export_local(
        &self,
        view: &ExportView<'_>,
        endpoint: &Arc<ExportTable>,
        next: &AtomicU64,
        bytes: ByteReservation,
        stream: impl Issue,
    ) -> Result<Locator> {
        let event = match producer(view.spans, stream.stream())? {
            Some(_) => Some(self.events.fence(stream, false)?),
            None => None,
        };
        let buffer = Arc::new(TransportBuffer::local(view.spans.clone(), event, bytes));
        let locator = view.locator(
            &self.source,
            TransferTransport::Local {
                endpoint: endpoint.name().to_owned(),
                key: next.fetch_add(1, Ordering::Relaxed),
            },
        );
        if let Err(error) = endpoint.register(locator.clone(), Arc::clone(&buffer)) {
            TransportBuffer::reclaim(buffer, &self.events, None)?;
            return Err(error);
        }
        Ok(locator)
    }
    pub(super) fn fetch_local(
        &self,
        locator: &Locator,
        device: MemoryKind,
        destination: Option<&Spans>,
        region: Option<&[Range<u64>]>,
        reservation: Option<&ReadReservation>,
        stream: impl Issue,
    ) -> Result<Arc<Read>> {
        if let Some(error) = self.reads.error() {
            return Err(Error::Shared(error));
        }
        let (buffer, grant) = ExportTable::acquire_read(locator)?;
        let borrowed = destination.is_none();
        let borrower = borrowed.then(|| Arc::downgrade(&self.borrowed));
        let read = Read::new(Arc::clone(&self.events), borrower);
        read.bind_source(grant)?;

        let submitted = (|| -> Result<Arc<Read>> {
            let source = buffer.spans()?;
            let producer = buffer.event();
            if source.device() != device {
                return Err(Error::Invalid(
                    "local binding requires the source device".into(),
                ));
            }
            let source = match region {
                Some(region) => {
                    within(region, &locator.shape)?;
                    source.region(region)?
                }
                None => source,
            };

            match destination {
                Some(destination) => {
                    let target = read_destination(locator, device, Some(destination), region)?;
                    let backend = NativeRead {
                        read: Arc::clone(&read),
                        call: ReadCall::Copy {
                            source,
                            destination: target,
                            producer,
                        },
                        context: Arc::clone(&self.context),
                    };
                    read::submit(&self.reads, backend, locator.nbytes, stream, reservation)
                }
                None => {
                    {
                        let mut reads = lock(&self.borrowed);
                        let reads = reads
                            .as_mut()
                            .ok_or(Error::Resource("transfer pool is closed"))?;
                        reads.insert(Arc::as_ptr(&read) as usize, Arc::downgrade(&read));
                    }
                    read::borrow(
                        &read,
                        &self.capacity,
                        source,
                        producer,
                        reservation,
                        self.reads.completion_wake(),
                    )?;
                    Ok(Arc::clone(&read))
                }
            }
        })();
        if submitted.is_err()
            && let Err(cleanup) = read.release_source()
        {
            tracing::warn!("failed to release a local transfer source: {cleanup}");
        }
        submitted
    }
}
