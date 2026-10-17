//! The rank-channel mechanism: the export's bytes travel in its location
//! over the head's channel. An export stages them on the host
//! ([`Staged::Channel`](super::transport::Staged)); its location is complete
//! once the copy into the staging was performed. A read copies them from
//! the location into its destination.
//!
//! A device product is staged in a page-locked block leased from the
//! transport's staging and copied into the location from there; a host
//! product is copied straight into the bytes that become the location's
//! payload.

use std::ops::Range;
use std::sync::Arc;

use uniserve_cuda::SourceOrder;
use uniserve_ipc::{Locator, TransferTransport};

use super::read::{NativeRead, ReadCall, payload};
use super::staging::Lease;
use super::transport::{ExportView, Staged, Transport, producer, read_destination};
use super::{Spans, copy_planner};
use crate::ffi::{Backing, MemoryKind};
use crate::storage::memory::StorageView;
use crate::stream::Issue;
use crate::{Error, Result};

/// A channel export's bytes until its location is complete.
pub(crate) enum ChannelBytes {
    /// A device product's bytes in a leased page-locked block.
    Pinned(Arc<Lease>),
    /// A host product's bytes, which become the location's payload.
    Host(Arc<Payload>),
}

/// The payload of a host product's channel location, written in place by
/// the export's host copy before the location takes it.
pub(crate) struct Payload {
    bytes: Vec<u8>,
    /// The bytes' address, taken once with write access for the copy.
    address: u64,
}

impl Payload {
    fn new(bytes: usize) -> Arc<Self> {
        let mut bytes = vec![0; bytes];
        let address = bytes.as_mut_ptr() as u64;
        Arc::new(Self { bytes, address })
    }
}

impl Backing for Payload {
    fn device(&self) -> MemoryKind {
        MemoryKind::Host
    }

    fn address(&self) -> u64 {
        self.address
    }

    fn bytes(&self) -> usize {
        self.bytes.len()
    }
}

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
        let (staging, backing): (ChannelBytes, Arc<dyn Backing>) = match producer {
            Some(device) => {
                let lease = self.context.staging.lease(device, bytes)?;
                let backing = Arc::clone(&lease) as Arc<dyn Backing>;
                (ChannelBytes::Pinned(lease), backing)
            }
            None => {
                let payload = Payload::new(bytes.max(1));
                let backing = Arc::clone(&payload) as Arc<dyn Backing>;
                (ChannelBytes::Host(payload), backing)
            }
        };
        let target = Spans::single(StorageView::contiguous(
            backing,
            0,
            view.spans.dtype(),
            &shape,
        )?)?;
        let mut plan = copy_planner(producer, SourceOrder::Stream)?;
        target.plan_copy(view.spans, &mut plan)?;
        // The plan holds addresses; the staged export holds the storage.
        drop(target);

        match producer {
            Some(_) => {
                // SAFETY: the staged export keeps the staging alive until
                // its bytes are read, behind the wait below.
                unsafe { stream.copy_strided(&plan) }?;
                // The bytes leave with the result: a downstream reader cannot
                // wait on this stream.
                let _range = crate::profiling::range(c"channel_export_synchronize", None);
                if let Err(error) = stream.synchronize() {
                    // The copy may still write the block: it never returns
                    // to the staging.
                    retain(staging);
                    return Err(error);
                }
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

/// Keep staging whose device copy has an unknown completion until process
/// teardown.
#[expect(
    clippy::mem_forget,
    reason = "unknown device completion must retain the block it writes"
)]
fn retain(staging: ChannelBytes) {
    std::mem::forget(staging);
}

/// Complete a staged channel export: its location with the bytes its copy
/// staged, once that copy and the wait behind it were performed. A leased
/// block returns to the staging once its bytes are copied out.
pub(super) fn locate(mut locator: Locator, staging: ChannelBytes, bytes: usize) -> Result<Locator> {
    let payload = match staging {
        // SAFETY: the export's copy and the wait behind it were performed,
        // so nothing writes the block any more.
        ChannelBytes::Pinned(lease) => unsafe { lease.host_bytes() }
            .get(..bytes)
            .ok_or_else(|| Error::Invariant("channel staging is smaller than its bytes".into()))?
            .to_vec(),
        ChannelBytes::Host(payload) => {
            let mut payload = Arc::try_unwrap(payload)
                .map_err(|_| Error::Invariant("a staged channel payload is shared".into()))?
                .bytes;
            payload.truncate(bytes);
            payload
        }
    };
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
