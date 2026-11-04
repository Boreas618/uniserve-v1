//! Transfer admission, transports, readable results, and physical retirement.
//!
//! A transfer moves Rust-owned storage between this main process and a peer:
//! `Spans` of `StorageView`s, ordered by the rank's `EventPool`. `Spans` and
//! the copy planning live here first; the transports, readable results,
//! export tables and cache import/export follow as they move from the retired
//! worker (`crates/worker/src/transfer`), with their operations unchanged.

pub(crate) mod backing;
pub(crate) mod cache_export;
pub use cache_export::CacheExport;
mod cache_import;
mod capacity;
mod channel;
mod export_table;
pub mod exports;
pub mod fetch;
pub(crate) mod links;
mod local;
mod pool;
mod posix_shm;
mod read;
mod registry;
pub mod spans;
pub(crate) mod staging;
mod ticket;
mod transport;
pub(crate) mod vmm;
pub(crate) mod warming;

pub(crate) use cache_import::{
    CacheImport, CacheImporter, ImportCopy, ImportEntry, ImportWorkspace, Retired,
};
pub use posix_shm::SharedRead;
pub(crate) use posix_shm::{SharedBuffer, SharedMapping};
pub use read::Read;
pub(crate) use read::ReadRef;
pub use spans::Spans;
pub(crate) use spans::{copy_on_host, copy_planner, submit_copy};
pub use transport::{Transport, TransportKind, TransportOptions, Transports};
pub(crate) use vmm::{DescriptorGrants, PeerStorage};

pub(crate) use capacity::ByteReservation;
pub use capacity::{ReadReservation, TransferCapacity};
pub(crate) use ticket::TransferTicket;

use std::ops::Range;
use std::sync::Arc;

use crate::completion::Completion;
use crate::{Error, Result};

/// A completion or readiness observer. Observers only notify (wake a
/// submission or the service loop); they never block or call back into the
/// owner that notifies them.
pub type Notify = Arc<dyn Fn() + Send + Sync>;

/// The physical retirement signal of an export, a read or a cache import.
pub type Retirement = Completion<Notify, ()>;

/// Run the observers of a completion, after its owner unlocked its state.
pub fn notify(observers: Vec<Notify>) {
    for observer in observers {
        observer();
    }
}

/// Reader acknowledgment words in the header of a shared-memory segment or a
/// pooled VMM chunk: one per consumer slot, which a reader sets to
/// [`CLAIMED`] before it copies and to [`ACKNOWLEDGED`] after; the producer
/// reuses the storage only once every claimed word is acknowledged.
pub(crate) const ACK_SLOTS: usize = 64;
pub(crate) const CLAIMED: u32 = 1;
pub(crate) const ACKNOWLEDGED: u32 = 2;

/// How long a host wait for device work may last where storage is given up
/// outside the rank's jobs: at shutdown, when no executor is left to route a
/// completion, and where a buffer closes before it was announced ready.
pub(crate) use crate::stream::events::{DRAIN_LIMIT, drain_event};

/// One source's contribution to a logical tensor fetch, in source-local and
/// destination-local coordinates.
pub(crate) struct ReadRegion {
    pub location: usize,
    pub source: Vec<Range<u64>>,
    pub destination: Vec<Range<u64>>,
}

/// Cover a requested region from bound locations in preference order.
///
/// Location bounds are already validated by their transfer descriptor. All
/// bounds use the logical tensor's axes. Missing coverage rejects the complete
/// plan, before any destination can be written. Return `None` for missing
/// coverage, and stop asking for locations as soon as coverage is complete.
/// An empty region needs no reads. Source lookup errors propagate unchanged.
pub(crate) fn plan_reads(
    region: &[Range<u64>],
    locations: impl IntoIterator<Item = Result<(usize, Vec<Range<u64>>)>>,
) -> Result<Option<Vec<ReadRegion>>> {
    if region.iter().any(Range::is_empty) {
        return Ok(Some(Vec::new()));
    }

    let mut missing = vec![region.to_vec()];
    let mut reads = Vec::new();
    for location in locations {
        let (location, covered) = location?;
        let mut remaining = Vec::new();
        for required in missing {
            if let Some(overlap) = ipc::subtract_region(&required, &covered, &mut remaining) {
                reads.push(ReadRegion {
                    location,
                    source: relative_region(&overlap, &covered),
                    destination: relative_region(&overlap, region),
                });
            }
        }

        if remaining.is_empty() {
            return Ok(Some(reads));
        }
        missing = remaining;
    }

    Ok(None)
}

fn relative_region(region: &[Range<u64>], origin: &[Range<u64>]) -> Vec<Range<u64>> {
    region
        .iter()
        .zip(origin)
        .map(|(axis, base)| axis.start - base.start..axis.end - base.start)
        .collect()
}
