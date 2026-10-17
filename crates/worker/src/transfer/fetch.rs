//! Plan a logical tensor's reads from its locations and submit them.

use crate::ffi::DType;
use std::ops::Range;
use std::sync::Arc;

use uniserve_ipc::{Locator, TensorTransfer};

use super::read::Read;
use super::transport::{Transport, TransportKind, Transports};
use super::{ReadReservation, Spans, plan_reads};
use crate::stream::Issue;
use crate::{Error, Result};

/// One source's read: the region of its locator (in locator coordinates)
/// and the destination piece it fills.
pub(crate) struct PlannedRead {
    transport: Arc<Transport>,
    locator: Arc<Locator>,
    source_region: Vec<Range<u64>>,
    target: Spans,
}

/// Cover `region` of `tensor` from its locations, preferring a
/// process-local replica and otherwise keeping export order, and pair each
/// covering read with its piece of `destination` (shaped like `region`).
///
/// The whole destination is validated, and missing coverage rejects the
/// plan, before any read is submitted. `locators` caches shared locators
/// per location index across calls on one descriptor, so channel payloads
/// are cloned once however many regions read them.
pub(crate) fn plan(
    tensor: &TensorTransfer,
    destination: &Spans,
    transports: &Transports,
    region: &[Range<u64>],
    locators: &mut [Option<Arc<Locator>>],
) -> Result<Vec<PlannedRead>> {
    let first = tensor
        .locations
        .first()
        .ok_or_else(|| Error::Invalid("tensor transfer has no locations".into()))?;
    let shape: Vec<u64> = region.iter().map(|range| range.end - range.start).collect();
    destination.validate_destination(&shape, DType::parse(&first.dtype)?, destination.device())?;

    let mut ordered: Vec<_> = tensor.locations.iter().enumerate().collect();
    ordered.sort_by_key(|(_, location)| TransportKind::of(location) != TransportKind::Local);
    let mut bound = Vec::new();
    let locations = ordered.into_iter().filter_map(|(index, location)| {
        // A POSIX segment is addressable only on its producer's node; other
        // locations of the same tensor may still cover the region.
        let transport = transports
            .get(TransportKind::of(location))
            .filter(|transport| transport.reads(location))?;
        let covered = location
            .offset
            .iter()
            .zip(&location.shape)
            .map(|(&start, &extent)| start..start + extent)
            .collect();
        let slot = bound.len();
        bound.push((Arc::clone(transport), index));
        Some(Ok((slot, covered)))
    });
    let reads = plan_reads(region, locations)?.ok_or_else(|| {
        Error::Invalid("bound product locations do not cover the consumer region".into())
    })?;

    reads
        .into_iter()
        .map(|read| {
            let (transport, index) = &bound[read.location];
            let locator = match &locators[*index] {
                Some(locator) => Arc::clone(locator),
                None => {
                    let locator = Arc::new(tensor.locations[*index].clone());
                    locators[*index] = Some(Arc::clone(&locator));
                    locator
                }
            };
            Ok(PlannedRead {
                transport: Arc::clone(transport),
                locator,
                source_region: read.source,
                target: destination.region(&read.destination)?,
            })
        })
        .collect()
}

/// Issue planned reads under one reservation of their whole fan-out.
/// `stream` orders the destination's earlier writes before the copies: each
/// read's handoff fence is issued to it. `retain` registers each read with
/// its storage owner as it is issued. A failure cancels the issued reads
/// without releasing their physical accesses; unused credits return when
/// the reservation closes.
///
/// The caller starts the returned reads ([`start`]) once `stream`
/// performed its work: at once after a stream, from the job's completion
/// on after a recorder.
pub(crate) fn submit(
    reads: &[PlannedRead],
    stream: impl Issue,
    mut retain: impl FnMut(&Arc<Read>) -> Result<()>,
) -> Result<Vec<Arc<Read>>> {
    let Some(first) = reads.first() else {
        return Ok(Vec::new());
    };
    let capacity = first.transport.capacity();
    if reads
        .iter()
        .any(|read| !Arc::ptr_eq(read.transport.capacity(), capacity))
    {
        return Err(Error::Invalid(
            "a fetch's transports must share the rank's transfer capacity".into(),
        ));
    }

    let reservation = ReadReservation::new(Arc::clone(capacity), reads.len())?;
    let mut submitted: Vec<Arc<Read>> = Vec::with_capacity(reads.len());
    let result = (|| -> Result<()> {
        for read in reads {
            let ticket = read.transport.issue_fetch(
                &read.locator,
                read.target.device(),
                Some(&read.target),
                Some(&read.source_region),
                Some(&reservation),
                stream,
            )?;
            submitted.push(Arc::clone(&ticket));
            retain(&ticket)?;
        }
        Ok(())
    })();

    if result.is_err() {
        for read in &submitted {
            if let Err(cleanup) = read.cancel() {
                tracing::warn!("failed to cancel a read of a failed fetch: {cleanup}");
            }
        }
    }
    let closed = reservation.close();
    result?;
    closed?;
    Ok(submitted)
}

/// Hand issued reads to their transport threads ([`Read::start`]).
pub fn start(reads: &[Arc<Read>]) -> Result<()> {
    reads.iter().try_for_each(|read| read.start())
}

/// Read `region` of `tensor` into `destination`: `plan` then `submit`,
/// whose caller starts the reads.
pub fn fetch_tensor(
    tensor: &TensorTransfer,
    destination: &Spans,
    transports: &Transports,
    region: &[Range<u64>],
    stream: impl Issue,
    retain: impl FnMut(&Arc<Read>) -> Result<()>,
) -> Result<Vec<Arc<Read>>> {
    let mut locators = vec![None; tensor.locations.len()];
    let reads = plan(tensor, destination, transports, region, &mut locators)?;
    submit(&reads, stream, retain)
}
