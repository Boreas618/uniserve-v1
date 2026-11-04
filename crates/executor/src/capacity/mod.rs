//! Device storage capacity: the rank's grant on a GPU, its graph storage
//! budget, and the startup checks that refuse a runtime that does not fit.
//!
//! A deployment may use `⌊total × fraction⌋` of a GPU (its quota), and every
//! world of the deployment that uses the GPU, with the host's Transfer
//! Server, holds its storage within that one quota: the Storage Server keeps
//! the GPU's ledger. A rank's grant is the quota less everything the ledger
//! books beside the rank's own process, and less what the process holds now.
//!
//! The plan that sizes the weights, the cache, the products and the graph
//! pools from those figures is [`plan`]; it takes the rank's launch
//! parameters as [`CapacityRequest`], which the engine builds from the launch
//! description it already holds.

pub mod checks;
pub mod device;
mod plan;

pub use plan::{CapacityRequest, MediaFacts, Plan, Report, TokenFacts, token_plan};

use ipc::Call;

use crate::{Error, Result};

/// The smallest cache unit pool a rank keeps when the grant would allow less.
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
    /// The units a configured token capacity takes
    /// (`CachePlan::token_units`), which size the pool instead of the
    /// storage grant.
    pub token_units: Option<u64>,
    /// The exact units of a standby worker's pool: those of the worker it
    /// stands by for, whose planes it maps rather than allocates.
    pub units: Option<u64>,
    /// Device bytes of one unit, with what other owners keep per unit.
    pub unit_bytes: u64,
    /// The device storage grant, if the rank has one.
    pub available_bytes: Option<u64>,
    /// Units' worth of the grant that other storage occupies.
    pub reserved_units: u64,
    /// Whether the rank's world succeeds another on its devices: its pool
    /// is laid out beside the predecessor's segments, and the part of a
    /// configured capacity its grant does not hold yet is created as the
    /// predecessor's storage frees (its deferred tail segments).
    pub successor: bool,
    /// The units of a configured token capacity that a successor world maps
    /// from its predecessor's segments rather than allocates: the grant pays
    /// for the rest alone, since the ledger books those once, for the
    /// predecessor.
    pub inherited_units: u64,
}

impl CacheCapacity {
    /// Explicit token capacity counts whole pages of every K/V group and a
    /// state checkpoint per request slot. A successor world's own units are
    /// those of the capacity past the ones it inherits, of which it creates
    /// now what its grant holds beside the reserved ones; the grant of any
    /// other world must hold the whole capacity.
    /// Automatic sizing gives the pool the complete units the grant holds
    /// beside the reserved ones; without a grant the pool takes
    /// [`DEFAULT_NUM_UNITS`]. Every mode leaves room for the sentinel unit
    /// and one page of every group.
    pub(crate) fn derive(demand: &CacheDemand<'_>) -> Result<Self> {
        let CacheDemand {
            pages,
            token_units,
            units: adopted,
            unit_bytes,
            available_bytes,
            reserved_units,
            successor,
            inherited_units,
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
        let num_units = if let Some(units) = token_units {
            if units == 0 {
                return Err(Error::Invalid(
                    "configured KV token capacity must be positive".into(),
                ));
            }
            if successor {
                // The units past those it inherits are its own; the grant
                // pays for what it creates of them now, and the rest of its
                // tail follows as the predecessor's storage frees.
                let own = units.saturating_sub(inherited_units);
                let num_units = match available_bytes {
                    Some(available) => {
                        own.min(granted(available, unit_bytes, reserved_units, minimum)?)
                    }
                    None => own,
                };
                return Ok(Self {
                    num_units: minimum.max(num_units),
                });
            }
            minimum.max(units)
        } else if let Some(available) = available_bytes {
            granted(available, unit_bytes, reserved_units, minimum)?
        } else {
            minimum.max(DEFAULT_NUM_UNITS)
        };

        if let Some(available) = available_bytes
            && (num_units + reserved_units) * unit_bytes > available
        {
            tracing::warn!(
                units = num_units,
                reserved_units,
                unit_bytes,
                available,
                "the configured KV capacity exceeds the device storage grant"
            );
            return Err(Error::Resource(
                "configured KV storage exceeds the device storage grant",
            ));
        }

        Ok(Self { num_units })
    }
}

/// The units of `unit_bytes` a grant of `available` bytes holds beside
/// `reserved_units`: fails, logging the figures, when they are fewer than
/// `minimum`, the sentinel unit and one page of every group.
fn granted(available: u64, unit_bytes: u64, reserved_units: u64, minimum: u64) -> Result<u64> {
    let units = (available / unit_bytes).saturating_sub(reserved_units);
    if units < minimum {
        tracing::warn!(
            units,
            minimum,
            reserved_units,
            unit_bytes,
            available,
            "the device storage grant cannot hold the KV pool"
        );
        return Err(Error::Resource(
            "device storage grant cannot hold the required KV pool",
        ));
    }
    Ok(units)
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
    /// Relay entries of every request slot: the most calls of one request
    /// whose scalar products the rank holds at once.
    pub relay_depth: usize,
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
    pub host_task_inflight: usize,
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
    /// The arena of a token worker: relay entries for every unresolved call of
    /// every request slot and the one before them, whose completion the
    /// oldest of them still reads on the device as its predicate, and one
    /// transfer ticket and one host-pool task per queued call (`depth`
    /// batches of up to `calls` calls are in flight at once, and a call holds
    /// at most one of each), each ticket budgeted for the largest single
    /// read.
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
        let relay_depth = window + 1;
        let device_product_bytes = relay_bytes(requests, relay_depth);

        Ok(Self {
            relay_depth,
            device_product_bytes,
            transfer_bytes: transfer_bytes_per_read.max(1) * transfer_tickets as u64,
            transfer_tickets,
            export_pool_bytes: if device_exports {
                export_pool_bytes(device_product_bytes)
            } else {
                0
            },
            host_task_inflight: transfer_tickets,
        })
    }

    /// A media rank retains every local product in full. It holds a read
    /// ticket per queued call or per concurrent artifact import of every
    /// slot, whichever is more, and at least one per rank of the
    /// deployment, since assembly reads from every producing rank,
    /// including stateless ranks.
    ///
    /// The engine queues a request's latent preparation and denoising steps
    /// without waiting for their resolution, each predicated on the
    /// completion of the one before it, so one request may hold a call in
    /// every batch in flight on the rank (`depth`, one call of a request per
    /// media batch). Every slot therefore has a relay entry per batch in
    /// flight, and one more for the completion the oldest of those calls
    /// still reads on the device.
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
        let relay_depth = depth + 1;
        let device_product_bytes = relay_bytes(requests, relay_depth);

        Ok(Self {
            relay_depth,
            device_product_bytes,
            transfer_bytes: (requests as u64 * product_bytes_per_request).max(1),
            transfer_tickets: ranks.max(slots),
            export_pool_bytes: if device_exports {
                export_pool_bytes(device_product_bytes)
            } else {
                0
            },
            host_task_inflight: requests * (window + 1),
        })
    }

    /// Device bytes the rank allocates on the primary device on first use,
    /// beside what startup allocates: the relay arenas and the export pool.
    pub(crate) fn deferred_device_bytes(&self) -> u64 {
        self.device_product_bytes + self.export_pool_bytes
    }
}

/// Bytes of the product store's relay arenas: `depth` entries for every
/// request slot and the padding slot zero. An entry holds one element of
/// every scalar a call relays (`Call::RELAY_DTYPES`), in arenas allocated per
/// dtype and field.
fn relay_bytes(requests: usize, depth: usize) -> u64 {
    ((requests + 1) * depth) as u64 * Call::relay_row_bytes()
}

/// The share of a device's bytes a rank may reserve for graph storage: the
/// capture pools of its bucket graphs and the scratch they allocate.
///
/// The figure is capacity planning, so it lives here and the arena that maps
/// the pools takes the bytes as an input. Ten percent of the device holds the
/// buckets of a causal language model and a media model of the slice with
/// room to spare; a rank that captures more books it as its own graph item,
/// so the ledger refuses it before the device runs out.
pub fn graph_storage_budget_bytes(total_device_bytes: i64) -> u64 {
    (total_device_bytes.max(0) as f64 * 0.10) as u64
}

/// A deployment's quota of a device of `total` bytes for `fraction` of it:
/// `⌊total × fraction⌋`.
pub fn quota(total: u64, fraction: f64) -> u64 {
    (total as f64 * fraction) as u64
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A pool demand of two groups with 64-token pages of one unit each and
    /// 1 KiB units.
    fn demand<'a>(pages: &'a [(u32, u32)]) -> CacheDemand<'a> {
        CacheDemand {
            pages,
            token_units: None,
            units: None,
            unit_bytes: 1024,
            available_bytes: Some(1024 * 100),
            reserved_units: 0,
            successor: false,
            inherited_units: 0,
        }
    }

    #[test]
    fn an_automatic_pool_takes_the_whole_units_its_grant_holds() -> Result<()> {
        let pages = [(64, 1), (64, 1)];
        let capacity = CacheCapacity::derive(&demand(&pages))?;
        // 100 units of 1 KiB fit; one page of every group plus the sentinel
        // stays free, so the pool is the rest.
        assert_eq!(capacity.num_units, 100 - 2 - 1);
        Ok(())
    }

    #[test]
    fn a_pool_without_a_grant_keeps_the_smallest_pool() -> Result<()> {
        let pages = [(64, 1), (64, 1)];
        let mut demand = demand(&pages);
        demand.available_bytes = None;
        assert_eq!(CacheCapacity::derive(&demand)?.num_units, DEFAULT_NUM_UNITS);
        Ok(())
    }

    #[test]
    fn an_explicit_token_capacity_counts_whole_pages_and_state_checkpoints() -> Result<()> {
        let pages = [(64, 1), (64, 1)];
        let mut demand = demand(&pages);
        demand.token_units = Some(640);
        // Ten pages of each group plus the sentinel.
        let capacity = CacheCapacity::derive(&demand)?;
        assert!(capacity.num_units >= 640);
        Ok(())
    }

    #[test]
    fn a_demand_without_pages_or_units_is_refused() {
        assert!(
            CacheCapacity::derive(&CacheDemand {
                pages: &[],
                token_units: None,
                units: None,
                unit_bytes: 1024,
                available_bytes: None,
                reserved_units: 0,
                successor: false,
                inherited_units: 0,
            })
            .is_err()
        );
        assert!(
            CacheCapacity::derive(&CacheDemand {
                pages: &[(0, 1)],
                token_units: None,
                units: None,
                unit_bytes: 1024,
                available_bytes: None,
                reserved_units: 0,
                successor: false,
                inherited_units: 0,
            })
            .is_err()
        );
    }

    #[test]
    fn a_quota_is_the_fraction_of_the_device_and_the_graph_share_is_its_tenth() {
        assert_eq!(quota(1000, 0.7), 700);
        assert_eq!(quota(0, 0.7), 0);
        assert_eq!(graph_storage_budget_bytes(1000), 100);
        assert_eq!(graph_storage_budget_bytes(-1), 0);
    }
}
