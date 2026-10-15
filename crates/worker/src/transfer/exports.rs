//! Export selection across a rank's transports.

use std::sync::Arc;

use uniserve_ipc::Locator;

use super::cuda_vmm::Exported;
use super::transport::{Staged, Transport, TransportKind, Transports};
use super::{Retirement, Spans};
use crate::ffi::Device;
use crate::stream::Issue;
use crate::{Error, Result};

/// One accepted export location and the transport that owns it.
pub type Location = (Arc<Transport>, Locator);

/// Export `source` through every export mechanism that reaches its
/// consumers: local plus CUDA VMM for device memory when the rank exports
/// over VMM (unless `host` forces host mechanisms), otherwise local, shared
/// memory and the rank channel.
///
/// `consumers` are the acknowledgment slots of the other ranks that read the
/// export. This rank reads its own exports through the local mechanism, which
/// is therefore always selected and reaches none of them.
///
/// The storage owner retains each location's retirement signal through
/// `retain` before the next export starts. A device payload that does not
/// fit its VMM pool falls back to the host mechanisms that reach its
/// consumers, and fails when it has consumers and none of those mechanisms
/// accepts it. Any failure revokes every accepted location; their physical
/// accesses still retire asynchronously.
///
/// The export's device work is issued to `stream` (see
/// [`Transport::export`]). Its locations are complete once that work was
/// performed: [`Exports::locate`] gives them then.
pub fn export(
    transports: &Transports,
    source: &Spans,
    offset: Option<&[u64]>,
    consumers: &[usize],
    host: bool,
    stream: impl Issue,
    mut retain: impl FnMut(Arc<Retirement>) -> Result<()>,
) -> Result<Exports> {
    if transports.exports_nothing() {
        return Err(Error::Unsupported(
            "tensor export requires a configured transport".into(),
        ));
    }

    let device = matches!(source.device(), Device::Cuda(_)) && !host;
    let kinds: &[TransportKind] = if device && transports.exporter(TransportKind::CudaVmm).is_some()
    {
        &[TransportKind::Local, TransportKind::CudaVmm]
    } else {
        &[
            TransportKind::Local,
            TransportKind::Shm,
            TransportKind::Channel,
        ]
    };
    let selected: Vec<_> = kinds
        .iter()
        .filter_map(|&kind| transports.exporter(kind))
        .filter(|transport| transport.serves(consumers))
        .cloned()
        .collect();

    let mut locations = Vec::new();
    let mut staged = Vec::new();
    // One attempt: a mechanism that fails gives its storage back, and the
    // locations accepted before it are revoked, so a job keeps none of the
    // work a failed export recorded.
    let result = stream.attempt(|| -> Result<()> {
        for transport in selected {
            match transport.export(source, offset, consumers, stream)? {
                Exported::Located(locator) => {
                    accept(&transport, *locator, &mut retain, &mut locations)?;
                }
                Exported::Staged(export) => staged.push((Arc::clone(&transport), *export)),
                Exported::PoolExhausted => {
                    // Local was selected before VMM, so never repeat it.
                    let mut carried = false;
                    for kind in [TransportKind::Shm, TransportKind::Channel] {
                        let Some(fallback) = transports.exporter(kind) else {
                            continue;
                        };
                        if !fallback.serves(consumers) {
                            continue;
                        }
                        match fallback.export(source, offset, consumers, stream)? {
                            Exported::Located(locator) => {
                                accept(fallback, *locator, &mut retain, &mut locations)?;
                                carried = true;
                            }
                            Exported::Staged(export) => {
                                staged.push((Arc::clone(fallback), *export));
                                carried = true;
                            }
                            Exported::PoolExhausted => {
                                return Err(Error::Invariant(
                                    "a host mechanism reported VMM exhaustion".into(),
                                ));
                            }
                        }
                    }
                    // The local location accepted earlier reaches no other
                    // rank: an export with consumers needs a mechanism that
                    // carries the payload to them.
                    if !carried && (locations.is_empty() || !consumers.is_empty()) {
                        return Err(Error::Resource(
                            "device product does not fit its VMM pool and the rank binds no \
                             other mechanism that exports it",
                        ));
                    }
                }
            }
        }
        Ok(())
    });

    if let Err(error) = result {
        // A failed attempt's work is not performed, so what it staged can
        // be given up at once.
        revoke(&locations);
        discard(staged);
        return Err(error);
    }
    Ok(Exports { locations, staged })
}

/// What an export accepted: its locations, and the channel exports whose
/// locations exist once their bytes were staged.
pub struct Exports {
    locations: Vec<Location>,
    staged: Vec<(Arc<Transport>, Staged)>,
}

impl Exports {
    /// Every location of the export, in the order its mechanisms were
    /// selected.
    ///
    /// The caller keeps this value until the issuer the export was given
    /// performed its work and calls this only then: at once after a stream,
    /// from the job's completion on after a recorder. Each staged
    /// location's retirement is retained through `retain`. A failure
    /// revokes every location.
    pub fn locate(
        self,
        mut retain: impl FnMut(Arc<Retirement>) -> Result<()>,
    ) -> Result<Vec<Location>> {
        let Self {
            mut locations,
            staged,
        } = self;
        let mut staged = staged.into_iter();
        let result = staged.by_ref().try_for_each(|(transport, export)| {
            accept(
                &transport,
                transport.locate(export)?,
                &mut retain,
                &mut locations,
            )
        });

        if let Err(error) = result {
            revoke(&locations);
            discard(staged);
            return Err(error);
        }
        Ok(locations)
    }

    /// Revoke the locations of an export that is not published and give up
    /// what it still stages. As [`Exports::locate`], call this once the
    /// issuer the export was given performed its work: the work addresses
    /// what this releases.
    pub fn revoke(self) {
        revoke(&self.locations);
        discard(self.staged);
    }
}

/// Give up the staged exports of an export that failed or is not
/// published; a failure is logged, the export's own outcome stands.
fn discard(staged: impl IntoIterator<Item = (Arc<Transport>, Staged)>) {
    for (transport, export) in staged {
        if let Err(cleanup) = transport.discard(export) {
            tracing::warn!(target: "uniserve_worker.transfer", "failed to give up a staged export: {cleanup}");
        }
    }
}

/// Revoke `locations` after a failed export, whose error stays the one
/// reported.
fn revoke(locations: &[Location]) {
    if let Err(cleanup) = release(locations) {
        tracing::warn!(target: "uniserve_worker.transfer", "failed to revoke an export after a failed export: {cleanup}");
    }
}

fn accept(
    transport: &Arc<Transport>,
    locator: Locator,
    retain: &mut impl FnMut(Arc<Retirement>) -> Result<()>,
    locations: &mut Vec<Location>,
) -> Result<()> {
    // Include this location in failure cleanup even when its retirement
    // lookup or the storage owner's retention fails.
    locations.push((Arc::clone(transport), locator.clone()));
    let retirement = transport.retirement(&locator)?;
    retain(retirement)
}

/// Revoke future reads of `locations`. Owners keep their backing until the
/// transports report physical retirement.
pub fn release(locations: &[Location]) -> Result<()> {
    let mut result = Ok(());
    for (transport, locator) in locations {
        result = result.and(transport.release(locator).map(drop));
    }
    result
}

#[cfg(test)]
mod tests {
    use std::collections::HashSet;

    use uniserve_ipc::{TransferTransport, WorkerEndpoint};

    use super::*;
    use crate::EventPool;
    use crate::ffi::{Backing, DType};
    use crate::lane::Job;
    use crate::storage::memory::{Memory, StorageView};
    use crate::transfer::{Capacity, TransportOptions, notify};

    /// A rank whose only mechanism towards slot 1 is the rank channel.
    fn channel() -> Result<Transports> {
        let capacity = Arc::new(Capacity::new(1 << 20, 8, notify)?);
        Transports::new(
            &[TransportKind::Local, TransportKind::Channel],
            capacity,
            EventPool::new(),
            &TransportOptions {
                source: WorkerEndpoint {
                    worker_id: "worker".into(),
                    rank: 0,
                    node: "node".into(),
                    address_space: "space".into(),
                    incarnation: "incarnation".into(),
                },
                acknowledgment_slot: 0,
                host_slots: HashSet::from([0]),
                cross_host_consumers: false,
                vmm_pool_bytes: 1 << 20,
            },
        )
    }

    /// A same-host reader waits on a CUDA VMM export through the handle of
    /// its readiness event, which exists once the event was recorded. A
    /// recorded export therefore receives it where it is located.
    #[test]
    #[ignore = "requires a CUDA device"]
    fn a_recorded_vmm_export_carries_its_event_once_the_job_recorded_it() -> Result<()> {
        use uniserve_cuda::{DeviceId, Stream};

        let stream = Stream::new(DeviceId(0))?;
        let capacity = Arc::new(Capacity::new(1 << 24, 8, notify)?);
        let transports = Transports::new(
            &[TransportKind::Local, TransportKind::CudaVmm],
            capacity,
            EventPool::new(),
            &TransportOptions {
                source: WorkerEndpoint {
                    worker_id: "worker".into(),
                    rank: 0,
                    node: "node".into(),
                    address_space: "space".into(),
                    incarnation: "incarnation".into(),
                },
                acknowledgment_slot: 0,
                host_slots: HashSet::from([0, 1]),
                cross_host_consumers: false,
                vmm_pool_bytes: 1 << 22,
            },
        )?;
        let source = Spans::single(StorageView::contiguous(
            Memory::device(0, 512)? as Arc<dyn Backing>,
            0,
            DType::F32,
            &[16, 8],
        )?)?;

        // Recording queues the readiness event's record; nothing is
        // recorded yet when the export returns.
        let job = Job::new();
        let exports = export(
            &transports,
            &source,
            None,
            &[1],
            false,
            job.on(Some(&stream)),
            |_| Ok(()),
        )?;
        job.perform().check().map_err(|failure| failure.error)?;

        let locations = exports.locate(|_| Ok(()))?;
        let handles: Vec<usize> = locations
            .iter()
            .filter_map(|(_, locator)| match &locator.transport {
                TransferTransport::CudaVmm {
                    ready_event_handle, ..
                } => Some(ready_event_handle.len()),
                _ => None,
            })
            .collect();
        assert_eq!(handles, [64]);
        release(&locations)?;
        stream.synchronize()?;
        Ok(())
    }

    #[test]
    fn a_recorded_channel_export_carries_what_its_job_copied() -> Result<()> {
        let transports = channel()?;
        let memory = Memory::host(4);
        let source = Spans::single(StorageView::contiguous(
            Arc::clone(&memory) as Arc<dyn Backing>,
            0,
            DType::U8,
            &[4],
        )?)?;

        // The export is recorded before its producer wrote the value, as a
        // forward records its publication behind its own operations.
        let job = Job::new();
        let exports = export(
            &transports,
            &source,
            None,
            &[1],
            false,
            job.on(None),
            |_| Ok(()),
        )?;
        // SAFETY: four bytes of host memory this test owns.
        unsafe {
            std::ptr::copy_nonoverlapping([1_u8, 2, 3, 4].as_ptr(), memory.address() as *mut u8, 4)
        };
        job.perform().check().map_err(|failure| failure.error)?;

        let locations = exports.locate(|_| Ok(()))?;
        let payloads: Vec<&[u8]> = locations
            .iter()
            .filter_map(|(_, locator)| match &locator.transport {
                TransferTransport::Channel { payload, .. } => Some(payload.as_slice()),
                _ => None,
            })
            .collect();
        assert_eq!(payloads, [[1_u8, 2, 3, 4].as_slice()]);
        Ok(())
    }
}
