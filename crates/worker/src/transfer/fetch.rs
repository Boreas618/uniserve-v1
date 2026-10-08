//! Plan a logical tensor's reads from its locations and submit them.

use std::ops::Range;
use std::sync::Arc;

use uniserve_cuda::Stream;
use uniserve_ipc::{Locator, TensorTransfer};

use super::read::Read;
use super::transport::{Transport, TransportKind, Transports};
use super::{ReadReservation, Spans, plan_reads};
use crate::storage::memory::parse_dtype;
use crate::{Error, Result};

/// One source's read: the region of its locator (in locator coordinates)
/// and the destination piece it fills.
pub struct PlannedRead {
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
pub fn plan(
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
    destination.validate_destination(&shape, parse_dtype(&first.dtype)?, destination.device())?;

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
        Some(Ok::<_, Error>((slot, covered)))
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

/// Submit planned reads under one reservation of their whole fan-out.
/// `stream` orders the destination's earlier writes before the copies.
/// `retain` registers each read with its storage owner as it is submitted.
/// A failure cancels the submitted reads without releasing their physical
/// accesses; unused credits return when the reservation closes.
pub fn submit(
    reads: &[PlannedRead],
    stream: Option<&Stream>,
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
            let ticket = read.transport.fetch(
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
                eprintln!("failed to cancel a read of a failed fetch: {cleanup}");
            }
        }
    }
    let closed = reservation.close();
    result?;
    closed?;
    Ok(submitted)
}

/// Read `region` of `tensor` into `destination`: [`plan`] then [`submit`].
pub fn fetch_tensor(
    tensor: &TensorTransfer,
    destination: &Spans,
    transports: &Transports,
    region: &[Range<u64>],
    stream: Option<&Stream>,
    retain: impl FnMut(&Arc<Read>) -> Result<()>,
) -> Result<Vec<Arc<Read>>> {
    let mut locators = vec![None; tensor.locations.len()];
    let reads = plan(tensor, destination, transports, region, &mut locators)?;
    submit(&reads, stream, retain)
}
