//! The GPUs' memory ledgers ([`GpuLedger`]): one per GPU for each deployment
//! that uses it.
//!
//! A ledger is derived on demand from three sources:
//!
//! - The device storages the server retains, each booked once in the ledger
//!   of every deployment that retains it, however many scopes and worlds map
//!   it: a read-only storage as [`Item::Weights`], one a client marked with
//!   its KV cache segment as [`Item::CacheSegment`], any other exclusive one
//!   as [`Item::Exclusive`]. An entry names the worlds that hold it, so a
//!   segment a successor world maps read-only appears once, held by both
//!   worlds, and becomes the successor's alone when the predecessor is
//!   retired: the commit's handover of the segments is a transfer on the
//!   ledger, never a release and an allocation.
//! - The CUDA context of the server's Transfer Server on each GPU, as it
//!   measured it when it started (`Request::Measured`), in every ledger of
//!   that GPU, for as long as it is connected.
//! - What the processes book themselves ([`Request::Book`]), which the server
//!   cannot see: a worker's graph pool, process memory and staging under its
//!   scope and world, the Transfer Server's staging. A booking lives as long
//!   as its process's connection, since the process's memory ends with the
//!   process. The processes of one scope and world serve in turn (a rank and
//!   the relaunch or standby that replaces it), so the largest of their
//!   bookings of an item counts.
//!
//! The quota of a deployment's ledger is `⌊total × fraction⌋` of the whole
//! GPU, which its workers state with their bookings; while two worlds that
//! state different fractions hold memory on the GPU, the smaller quota
//! holds. Every change that adds to a ledger, a booking or a new device
//! storage, is refused when it would take the ledger past its quota, so a
//! ledger's sum never exceeds it. The Transfer Server's context, whose
//! memory exists by the time it reports it, is booked as measured: the
//! worlds size themselves beside it.
//!
//! [`Request::Book`]: storage::protocol::Request::Book

use std::collections::{BTreeMap, BTreeSet};

use serde::{Deserialize, Serialize};
use storage::protocol::{
    Access, GpuLedger, Item, LedgerEntry, Retainer, Scope, Segment, StorageRecord,
};

use crate::ClientId;

/// What books an item: a worker of a scope and world, or the Transfer
/// Server.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub(crate) enum Booker {
    Worker { scope: Scope, world: u64 },
    Transfer,
}

/// One process's bookings.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub(crate) struct Booking {
    pub(crate) booker: Booker,
    /// Bytes per GPU and item; an item booked at 0 is absent.
    pub(crate) items: BTreeMap<([u8; 16], Item), u64>,
}

/// One retained device storage as the ledgers see it: its record, whether
/// its tensors are read-only, and the scopes of the worlds that retain any
/// of them.
pub(crate) struct Held<'a> {
    pub(crate) record: &'a StorageRecord,
    pub(crate) access: Access,
    pub(crate) retainers: BTreeSet<&'a Retainer>,
}

/// The ledgers' own state: every GPU's total memory, the quotas the worlds
/// state and the processes' bookings. The storages they derive the rest
/// from are the server's.
#[derive(Debug, Default, Clone, Serialize, Deserialize)]
pub(crate) struct Ledgers {
    /// Each GPU's total memory, as the last process that booked on it
    /// stated it.
    totals: BTreeMap<[u8; 16], u64>,
    /// The quota each world of a deployment states for a GPU, by
    /// `(deployment, GPU, world)`.
    quotas: BTreeMap<(String, [u8; 16], u64), u64>,
    /// The bookings of each connected process.
    bookings: BTreeMap<ClientId, Booking>,
}

/// An entry's identity within a ledger, under which its bytes add up.
type EntryKey = (Item, Option<Scope>, Vec<u64>, Option<Segment>);

impl Ledgers {
    /// Applies `client`'s booking on `gpu` and returns the state before it,
    /// for the caller to restore when it refuses the result.
    pub(crate) fn book(
        &mut self,
        client: ClientId,
        booker: Booker,
        gpu: [u8; 16],
        total: u64,
        quota: Option<u64>,
        items: &[(Item, u64)],
    ) -> Self {
        let before = self.clone();
        if total > 0 {
            self.totals.insert(gpu, total);
        }
        if let (Booker::Worker { scope, world }, Some(quota)) = (&booker, quota) {
            self.quotas
                .insert((scope.deployment.clone(), gpu, *world), quota);
        }
        let booking = self.bookings.entry(client).or_insert_with(|| Booking {
            booker,
            items: BTreeMap::new(),
        });
        for &(item, bytes) in items {
            if bytes == 0 {
                booking.items.remove(&(gpu, item));
            } else {
                booking.items.insert((gpu, item), bytes);
            }
        }
        before
    }

    /// The bookings of `client` end: its process is gone.
    pub(crate) fn disconnect(&mut self, client: ClientId) {
        self.bookings.remove(&client);
    }

    /// The quotas the worlds of `retired` stated end with their memory.
    pub(crate) fn retire(&mut self, deployment: &str, retired: &BTreeSet<u64>) {
        self.quotas
            .retain(|(of, _, world), _| of != deployment || !retired.contains(world));
    }

    /// Every quota of `deployment` ends: it is no longer served.
    pub(crate) fn release(&mut self, deployment: &str) {
        self.quotas.retain(|(of, _, _), _| of != deployment);
    }

    /// The ledgers of the GPUs `gpus` admits and the deployments `deployment`
    /// admits, derived from the retained device storages `held`, the
    /// Transfer Server's CUDA `contexts` (the bytes it measured on each GPU)
    /// and the bookings, sorted by GPU and deployment. A GPU that only the
    /// Transfer Server uses has a ledger of no deployment.
    pub(crate) fn ledgers<'a>(
        &self,
        held: impl IntoIterator<Item = Held<'a>>,
        contexts: &[([u8; 16], u64)],
        gpus: &dyn Fn(&[u8; 16]) -> bool,
        deployment: &dyn Fn(&str) -> bool,
    ) -> Vec<GpuLedger> {
        let mut books: BTreeMap<(String, [u8; 16]), BTreeMap<EntryKey, u64>> = BTreeMap::new();

        // Each retained storage once per deployment that retains it.
        for storage in held {
            let storage::protocol::Placement::Device { gpu } = storage.record.placement else {
                continue;
            };
            if !gpus(&gpu) {
                continue;
            }
            let item = match (storage.access, storage.record.segment) {
                (Access::ReadOnly, _) => Item::Weights,
                (Access::Exclusive, Some(_)) => Item::CacheSegment,
                (Access::Exclusive, None) => Item::Exclusive,
            };
            let mut by_deployment: BTreeMap<&str, Vec<&Retainer>> = BTreeMap::new();
            for retainer in &storage.retainers {
                by_deployment
                    .entry(retainer.scope.deployment.as_str())
                    .or_default()
                    .push(retainer);
            }
            for (name, retainers) in by_deployment {
                if !deployment(name) {
                    continue;
                }
                let scopes: BTreeSet<&Scope> =
                    retainers.iter().map(|retainer| &retainer.scope).collect();
                let scope = match scopes.into_iter().collect::<Vec<_>>().as_slice() {
                    [scope] => Some((*scope).clone()),
                    _ => None,
                };
                let worlds: Vec<u64> = retainers
                    .iter()
                    .map(|retainer| retainer.world)
                    .collect::<BTreeSet<_>>()
                    .into_iter()
                    .collect();
                let segment = storage
                    .record
                    .segment
                    .filter(|_| item == Item::CacheSegment);
                *books
                    .entry((name.to_owned(), gpu))
                    .or_default()
                    .entry((item, scope, worlds, segment))
                    .or_default() += storage.record.bytes;
            }
        }

        // Each worker's booking, the largest of its scope and world's
        // processes; the Transfer Server's, the largest of its processes.
        let mut workers: BTreeMap<(String, [u8; 16], EntryKey), u64> = BTreeMap::new();
        let mut transfer: BTreeMap<([u8; 16], Item), u64> = contexts
            .iter()
            .filter(|(gpu, bytes)| gpus(gpu) && *bytes > 0)
            .map(|&(gpu, bytes)| ((gpu, Item::TransferContext), bytes))
            .collect();
        for booking in self.bookings.values() {
            for (&(gpu, item), &bytes) in &booking.items {
                if !gpus(&gpu) {
                    continue;
                }
                match &booking.booker {
                    Booker::Worker { scope, world } => {
                        if !deployment(&scope.deployment) {
                            continue;
                        }
                        let key = (item, Some(scope.clone()), vec![*world], None);
                        let counted = workers
                            .entry((scope.deployment.clone(), gpu, key))
                            .or_default();
                        *counted = (*counted).max(bytes);
                    }
                    Booker::Transfer => {
                        let counted = transfer.entry((gpu, item)).or_default();
                        *counted = (*counted).max(bytes);
                    }
                }
            }
        }
        for ((name, gpu, key), bytes) in workers {
            *books
                .entry((name, gpu))
                .or_default()
                .entry(key)
                .or_default() += bytes;
        }

        // A deployment that stated a quota for a GPU has a ledger there even
        // before it holds anything.
        for (name, gpu, _) in self.quotas.keys() {
            if gpus(gpu) && deployment(name) {
                books.entry((name.clone(), *gpu)).or_default();
            }
        }

        // The Transfer Server's items are in every ledger of their GPU; a
        // GPU no deployment books has a ledger of its own for them.
        let transfer_gpus: BTreeSet<[u8; 16]> = transfer.keys().map(|(gpu, _)| *gpu).collect();
        for gpu in transfer_gpus {
            let mut names: Vec<String> = books
                .keys()
                .filter(|(_, of)| *of == gpu)
                .map(|(name, _)| name.clone())
                .collect();
            if names.is_empty() && deployment("") {
                names.push(String::new());
            }
            for name in names {
                let entries = books.entry((name, gpu)).or_default();
                for (&(of, item), &bytes) in &transfer {
                    if of == gpu {
                        *entries.entry((item, None, Vec::new(), None)).or_default() += bytes;
                    }
                }
            }
        }

        books
            .into_iter()
            .map(|((name, gpu), entries)| GpuLedger {
                gpu,
                quota: self
                    .quotas
                    .iter()
                    .filter(|((of, at, _), _)| *of == name && *at == gpu)
                    .map(|(_, quota)| *quota)
                    .min(),
                deployment: name,
                total: self.totals.get(&gpu).copied().unwrap_or(0),
                entries: entries
                    .into_iter()
                    .map(|((item, scope, worlds, segment), bytes)| LedgerEntry {
                        item,
                        scope,
                        worlds,
                        segment,
                        bytes,
                    })
                    .collect(),
            })
            .collect()
    }
}

/// The first ledger of `after` that a change takes past its quota, beyond
/// what it booked in `before`: a ledger already past its quota (as one whose
/// worlds stated a smaller fraction since) may still shrink, but not grow.
/// Returns the refusal's message, which names every entry of the ledger.
pub(crate) fn overrun(before: &[GpuLedger], after: &[GpuLedger]) -> Option<String> {
    after.iter().find_map(|ledger| {
        let quota = ledger.quota?;
        let used = ledger.used();
        let previous = before
            .iter()
            .find(|earlier| earlier.gpu == ledger.gpu && earlier.deployment == ledger.deployment)
            .map_or(0, GpuLedger::used);
        (used > quota && used > previous).then(|| {
            format!(
                "the memory ledger of GPU {} would book {used} bytes, past its quota of {quota} \
                 bytes by {}:\n{}",
                storage::protocol::gpu_name(&ledger.gpu),
                used - quota,
                ledger.describe()
            )
        })
    })
}
