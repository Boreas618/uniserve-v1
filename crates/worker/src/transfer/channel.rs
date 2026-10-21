//! The rank-channel mechanism: the export's bytes travel in its location
//! over the head's channel. An export stages them on the host
//! ([`Staged::Channel`](super::transport::Staged)); its location is complete
//! once the copy into the staging was performed. A read copies them from
//! the location into its destination.

use std::ops::Range;
use std::sync::Arc;

use cuda::SourceOrder;
use ipc::{Locator, TransferTransport};

use super::read::{NativeRead, ReadCall, payload};
use super::transport::{ExportView, Staged, Transport, producer, read_destination};
use super::{Spans, copy_planner};
use crate::ffi::{Backing, MemoryKind};
use crate::memory::allocation::{Storage, StorageView};
use crate::stream::Issue;
use crate::{Error, Result};

impl Transport {
    pub(super) fn export_channel(
        &self,
        view: &ExportView<'_>,
        name: &str,
        stream: impl Issue,
    ) -> Result<Staged> {
        let shape: Vec<i64> = view.shape.iter().map(|&extent| extent as i64).collect();
        let bytes = view.nbytes as usize;
        let producer = producer(view.spans, stream.stream())?.map(|(device, _)| device);
        let staging = match producer {
            Some(device) => Storage::pinned(device, bytes.max(1))?,
            None => Storage::host(bytes.max(1)),
        };
        let target = Spans::single(StorageView::contiguous(
            Arc::clone(&staging) as Arc<dyn Backing>,
            0,
            view.spans.dtype(),
            &shape,
        )?)?;
        let mut plan = copy_planner(producer, SourceOrder::Stream)?;
        target.plan_copy(view.spans, &mut plan)?;

        match producer {
            Some(_) => {
                // SAFETY: the staged export keeps the staging alive until
                // its bytes are read, behind the wait below.
                unsafe { stream.copy_strided(&plan) }?;
                // The bytes leave with the result: a downstream reader cannot
                // wait on this stream.
                let _range = crate::profiling::range(c"channel_export_synchronize", None);
                stream.synchronize()?;
            }
            // SAFETY: host operands owned by this export, written in the
            // producer's order.
            None => unsafe { stream.host().copy_strided(&plan) }?,
        }

        Ok(Staged::Channel {
            locator: view.locator(
                &self.source,
                TransferTransport::Channel {
                    endpoint: name.to_owned(),
                    payload: Vec::new(),
                },
            ),
            staging,
            bytes,
        })
    }
}

/// Complete a staged channel export: its location with the bytes its copy
/// staged, once that copy and the wait behind it were performed.
pub(super) fn locate(mut locator: Locator, staging: &Storage, bytes: usize) -> Result<Locator> {
    // SAFETY: the export's copy and the wait behind it were performed, so
    // nothing writes the staging any more.
    let payload = unsafe { staging.host_bytes() }
        .ok_or_else(|| Error::Invariant("channel staging is host memory".into()))?[..bytes]
        .to_vec();
    match &mut locator.transport {
        TransferTransport::Channel {
            payload: carried, ..
        } => *carried = payload,
        _ => {
            return Err(Error::Invariant(
                "a staged channel export has a channel location".into(),
            ));
        }
    }
    Ok(locator)
}

/// The read of a channel export's bytes into `destination` (see
/// [`Transport::issue_fetch`]).
pub(super) fn read(
    locator: &Locator,
    device: MemoryKind,
    destination: Option<&Spans>,
    region: Option<&[Range<u64>]>,
) -> Result<ReadCall> {
    Ok(ReadCall::Channel {
        locator: Arc::new(locator.clone()),
        destination: read_destination(locator, device, destination, region)?,
        region: region.map(<[_]>::to_vec),
    })
}

impl NativeRead {
    pub(super) fn channel(
        &self,
        locator: &Arc<Locator>,
        destination: &Spans,
        region: Option<&[Range<u64>]>,
    ) -> Result<()> {
        let TransferTransport::Channel { payload: bytes, .. } = &locator.transport else {
            return Err(Error::Invalid(
                "channel read requires a channel locator".into(),
            ));
        };
        if bytes.len() as u64 != locator.nbytes {
            return Err(Error::Invalid(
                "channel payload disagrees with its locator size".into(),
            ));
        }
        let backing: Arc<dyn Backing> = Arc::new(ChannelPayload(Arc::clone(locator)));
        let source = payload(backing, 0, locator)?;
        let source = match region {
            Some(region) => source.region(region)?,
            None => source,
        };
        self.copy_from_host(&source, destination, || {})
    }
}

/// A rank-channel payload as host memory a copy can read in place. The
/// locator is immutable and retained by the read for the copy's duration.
struct ChannelPayload(Arc<Locator>);

impl ChannelPayload {
    fn bytes(&self) -> &[u8] {
        match &self.0.transport {
            TransferTransport::Channel { payload, .. } => payload,
            _ => &[],
        }
    }
}

impl Backing for ChannelPayload {
    fn device(&self) -> MemoryKind {
        MemoryKind::Host
    }

    fn address(&self) -> u64 {
        self.bytes().as_ptr() as u64
    }

    fn bytes(&self) -> usize {
        self.bytes().len()
    }
}
