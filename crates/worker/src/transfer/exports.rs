//! Export selection across a rank's transports.

use std::sync::Arc;

use uniserve_cuda::Stream;
use uniserve_ipc::Locator;

use super::cuda_vmm::Exported;
use super::transport::{Transport, TransportKind, Transports};
use super::{Retirement, Spans};
use crate::ffi::Device;
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
pub fn export(
    transports: &Transports,
    source: &Spans,
    offset: Option<&[u64]>,
    consumers: &[usize],
    host: bool,
    stream: Option<&Stream>,
    mut retain: impl FnMut(Arc<Retirement>) -> Result<()>,
) -> Result<Vec<Location>> {
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
    let result = (|| -> Result<()> {
        for transport in selected {
            match transport.export(source, offset, consumers, stream)? {
                Exported::Located(locator) => {
                    accept(&transport, *locator, &mut retain, &mut locations)?;
                }
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
    })();

    if let Err(error) = result {
        for (transport, locator) in &locations {
            if let Err(cleanup) = transport.release(locator) {
                tracing::warn!(target: "uniserve_worker.transfer", "failed to revoke an export after a failed export: {cleanup}");
            }
        }
        return Err(error);
    }
    Ok(locations)
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
