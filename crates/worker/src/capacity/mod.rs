//! Startup storage bounds shared by model workers and their resource owners.
//!
//! [`plan`] resolves a rank's capacity after loading; [`device`] measures
//! device storage; [`checks`] verifies auxiliary devices and the final
//! startup footprint against the grant.

pub(crate) mod checks;
pub(crate) mod device;
pub(crate) mod plan;
pub(crate) mod video;

use ipc::Call;

use crate::{Error, Result};

pub(crate) const DEFAULT_NUM_UNITS: u64 = 4096;

/// The physical unit pool that fits beside a rank's fixed allocations.
#[derive(Clone, Copy, Debug)]
pub(crate) struct CacheCapacity {
    pub num_units: u64,
}

/// What a rank's cache pool is sized from.
#[derive(Clone, Copy, Debug)]
pub(crate) struct CacheDemand<'a> {
    /// `(tokens per page, units per page)` of every cache group.
    pub pages: &'a [(u32, u32)],
    /// A configured token capacity, which sizes the pool instead of the
    /// storage grant.
    pub tokens: Option<u64>,
    /// The exact units of a standby worker's pool: those of the worker it
    /// stands by for, whose planes it maps rather than allocates.
    pub units: Option<u64>,
    /// Device bytes of one unit, with what other owners keep per unit.
    pub unit_bytes: u64,
    /// The device storage grant, if the rank has one.
    pub available_bytes: Option<u64>,
    /// Units' worth of the grant that other storage occupies.
    pub reserved_units: u64,
}

impl CacheCapacity {
    /// Explicit token capacity counts whole pages of every cache group.
    /// Automatic sizing gives the pool the complete units the grant holds
    /// beside the reserved ones; without a grant the pool takes
    /// [`DEFAULT_NUM_UNITS`]. Every mode leaves room for the sentinel unit
    /// and one page of every group.
    pub(crate) fn derive(demand: &CacheDemand<'_>) -> Result<Self> {
        let CacheDemand {
            pages,
            tokens,
            units: adopted,
            unit_bytes,
            available_bytes,
            reserved_units,
        } = *demand;
        if pages.is_empty()
            || pages
                .iter()
                .any(|&(tokens, units)| tokens == 0 || units == 0)
            || unit_bytes == 0
        {
            return Err(Error::Invalid(
                "KV capacity dimensions must be positive".into(),
            ));
        }

        let minimum = 1 + pages
            .iter()
            .map(|&(_, units)| u64::from(units))
            .sum::<u64>();
        if let Some(units) = adopted {
            if units < minimum {
                return Err(Error::Invalid(
                    "an adopted KV pool holds fewer units than one page of every group".into(),
                ));
            }
            // The planes exist already: the grant need not hold them.
            return Ok(Self { num_units: units });
        }
        let num_units = if let Some(tokens) = tokens {
            if tokens == 0 {
                return Err(Error::Invalid(
                    "configured KV token capacity must be positive".into(),
                ));
            }
            minimum.max(units_for_tokens(pages, tokens))
        } else if let Some(available) = available_bytes {
            let units = (available / unit_bytes).saturating_sub(reserved_units);
            if units < minimum {
                return Err(Error::Resource(
                    "device storage grant cannot hold the required KV pool",
                ));
            }
            units
        } else {
            minimum.max(DEFAULT_NUM_UNITS)
        };

        if available_bytes
            .is_some_and(|available| (num_units + reserved_units) * unit_bytes > available)
        {
            return Err(Error::Resource(
                "configured KV storage exceeds the device storage grant",
            ));
        }

        Ok(Self { num_units })
    }
}

/// Count whole token pages across all cache groups, rounding down per group.
pub(crate) fn units_for_tokens(pages: &[(u32, u32)], tokens: u64) -> u64 {
    pages
        .iter()
        .map(|&(page, units)| tokens / u64::from(page) * u64::from(units))
        .sum()
}

/// Unresolved calls of a token worker, including a producer and its consumer
/// when a depth-one batch can contain both.
pub(crate) fn call_window(depth: usize, calls: usize) -> Result<usize> {
    if depth == 0 || calls == 0 {
        return Err(Error::Invalid(
            "call-window sizing requires positive bounds".into(),
        ));
    }

    Ok((depth * calls).min(depth.max(2)))
}

/// Media slots keep two unresolved outputs and one position for retirement.
pub(crate) fn request_tensor_window(depth: usize, slots: usize) -> Result<usize> {
    if slots == 0 || depth / slots < 3 {
        return Err(Error::Invalid(
            "request tensor pipeline requires two unresolved outputs per slot".into(),
        ));
    }

    Ok(depth / slots - 1)
}

/// Storage and concurrency bounds consumed by the worker's physical owners.
#[derive(Clone, Copy, Debug)]
pub(crate) struct ArenaCapacity {
    /// Device bytes the product store's relay arenas may take. The store's
    /// other products lie in the scheduler-placed buffer pool, which bounds
    /// them.
    pub device_product_bytes: u64,
    /// The rank's transfer byte budget: bytes every exported product and
    /// every read hold until they retire, whether or not they occupy storage
    /// of their own. A product published through several mechanisms holds
    /// its bytes once.
    pub transfer_bytes: u64,
    pub transfer_tickets: usize,
    /// Device bytes of the CUDA VMM export pool of the primary device, where
    /// the relay arenas lie; zero for a rank that exports no device product
    /// over CUDA VMM ([`export_pool_bytes`]).
    pub export_pool_bytes: u64,
    pub host_lane_inflight: usize,
}

/// Device bytes of a CUDA VMM export pool beside `relay_bytes` of relay
/// arenas.
///
/// The pool holds copies of the device products that do not lie in an
/// exportable allocation. Cache pages, frozen cache exports, buffer-pool
/// products and latent pages are exportable allocations, exported where they
/// lie, so the pool never holds them and does not grow with them. What
/// remains are the product store's relay scalars: at least one byte and at
/// most eight each, so `relay_bytes` bounds how many are live, and each
/// live scalar has at most one export, copied into a chunk of its own.
pub(crate) fn export_pool_bytes(relay_bytes: u64) -> u64 {
    crate::transfer::vmm::pool::pool_bytes(relay_bytes, size_of::<i64>())
}

/// The batching bounds of a token worker's arena.
#[derive(Clone, Copy, Debug)]
pub(crate) struct TokenBounds {
    /// Submitted batches the rank holds at once.
    pub depth: usize,
    /// Calls per batch.
    pub calls: usize,
    /// Request slots.
    pub requests: usize,
    /// Bytes one read of a product carries, at least one.
    pub transfer_bytes_per_read: u64,
    /// Whether the rank exports device products over CUDA VMM.
    pub device_exports: bool,
}

/// The batching bounds of a media worker's arena.
#[derive(Clone, Copy, Debug)]
pub(crate) struct RequestBounds {
    /// Submitted batches the rank holds at once.
    pub depth: usize,
    /// Calls per batch.
    pub calls: usize,
    /// Request slots.
    pub requests: usize,
    /// Ranks of the deployment, every one of which may read this rank's
    /// products.
    pub ranks: usize,
    /// Bytes of the products one request retains on this rank.
    pub product_bytes_per_request: u64,
    /// Artifact imports one request may have in flight.
    pub concurrent_imports: usize,
    /// Whether the rank exports device products over CUDA VMM.
    pub device_exports: bool,
}

impl ArenaCapacity {
    /// The arena of a token worker: relay lanes for every unresolved call
    /// of every request slot, and one transfer ticket and one host-lane task
    /// per queued call (`depth` batches of up to `calls` calls are in flight
    /// at once, and a call holds at most one of each), each ticket budgeted
    /// for the largest single read.
    pub(crate) fn for_tokens(bounds: &TokenBounds) -> Result<Self> {
        let TokenBounds {
            depth,
            calls,
            requests,
            transfer_bytes_per_read,
            device_exports,
        } = *bounds;
        let window = call_window(depth, calls)?;
        let transfer_tickets = depth * calls;
        let device_product_bytes = relay_bytes(requests, window);

        Ok(Self {
            device_product_bytes,
            transfer_bytes: transfer_bytes_per_read.max(1) * transfer_tickets as u64,
            transfer_tickets,
            export_pool_bytes: if device_exports {
                export_pool_bytes(device_product_bytes)
            } else {
                0
            },
            host_lane_inflight: transfer_tickets,
        })
    }

    /// A media rank retains every local product in full. It holds a read
    /// ticket per queued call or per concurrent artifact import of every
    /// slot, whichever is more, and at least one per rank of the
    /// deployment, since assembly reads from every producing rank,
    /// including stateless ranks.
    pub(crate) fn for_requests(bounds: &RequestBounds) -> Result<Self> {
        let RequestBounds {
            depth,
            calls,
            requests,
            ranks,
            product_bytes_per_request,
            concurrent_imports,
            device_exports,
        } = *bounds;
        let window = request_tensor_window(depth, requests)?;
        let slots = (depth * calls).max(requests * concurrent_imports);
        let device_product_bytes = relay_bytes(requests, window);

        Ok(Self {
            device_product_bytes,
            transfer_bytes: (requests as u64 * product_bytes_per_request).max(1),
            transfer_tickets: ranks.max(slots),
            export_pool_bytes: if device_exports {
                export_pool_bytes(device_product_bytes)
            } else {
                0
            },
            host_lane_inflight: requests * (window + 1),
        })
    }

    /// Device bytes the rank allocates on the primary device on first use,
    /// beside what startup allocates: the relay arenas and the export pool.
    pub(crate) fn deferred_device_bytes(&self) -> u64 {
        self.device_product_bytes + self.export_pool_bytes
    }
}

/// Bytes of the product store's relay arenas: one lane per unresolved call
/// and one retirement lane, for every request slot and the padding slot
/// zero. A lane holds one element of every scalar a call relays
/// (`Call::RELAY_DTYPES`), in arenas allocated per dtype and field.
fn relay_bytes(requests: usize, window: usize) -> u64 {
    ((requests + 1) * (window + 1)) as u64 * Call::relay_row_bytes()
}
