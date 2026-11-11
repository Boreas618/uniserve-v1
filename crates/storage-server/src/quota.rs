//! The GPUs' quotas ([`Quota`]): one per GPU for each deployment that uses
//! it.
//!
//! A quota's reservations are derived on demand from two sources:
//!
//! - The device storages the server retains, each reserved once against the
//!   quota of every deployment that retains it, however many scopes and
//!   epochs map it: a read-only storage as [`Item::Weights`], one a client
//!   marked with its KV cache segment as [`Item::CacheSegment`], any other
//!   exclusive one as [`Item::Exclusive`]. A reservation names the epochs
//!   that hold it, so a segment a successor epoch maps read-only appears
//!   once, held by both epochs, and becomes the successor's alone when the
//!   predecessor is retired: the commit's handover of the segments moves
//!   the reservation, and never releases and allocates.
//! - What the processes reserve themselves ([`Request::Quota`]), which the
//!   server cannot see: a worker's graph pool, process storage and staging
//!   under its scope and epoch; the Transfer Server's staging, and its CUDA
//!   context while it holds the context on the GPU, against every quota of
//!   the GPU. The processes of one scope and epoch serve in turn (a rank
//!   and the relaunch or standby that replaces it), so the largest of their
//!   reservations of an item counts. A worker that ends in order releases
//!   its reservations first; a worker's process that is lost leaves them
//!   kept for the process that replaces it, which takes each item over as
//!   it reserves it, so a successor epoch cannot take a lost rank's storage
//!   before its relaunch does. A kept reservation ends with its epoch's
//!   retirement or the deployment. The Transfer Server's reservations end
//!   with its connection.
//!
//! The limit of a deployment's quota is `⌊total × fraction⌋` of the whole
//! GPU, which its workers state with their reservations; while two epochs
//! that state different fractions hold storage on the GPU, the smaller
//! limit holds. Every change that adds to a quota's usage, a reservation or
//! a new device storage, is refused when it would take the usage past the
//! limit, so the usage never exceeds it. The Transfer Server reserves its
//! context before it retains the context, so a quota without room for it
//! refuses the context and the work that needed it.
//!
//! [`Request::Quota`]: ::storage::protocol::Request::Quota

use std::collections::{BTreeMap, BTreeSet};

use ::storage::protocol::{
    Access, Item, Quota, Reservation, Retainer, Scope, Segment, StorageRecord,
};
use serde::{Deserialize, Serialize};

use crate::ClientId;

/// What reserves an item: a worker of a scope and epoch, or the Transfer
/// Server.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub(crate) enum Reserver {
    Worker { scope: Scope, epoch: u64 },
    Transfer,
}

/// Bytes per GPU and item; an item reserved at 0 is absent.
pub(crate) type ItemBytes = BTreeMap<([u8; 16], Item), u64>;

/// One retained device storage as a quota counts it: its record, whether
/// its tensors are read-only, and the scopes of the epochs that retain any
/// of them.
pub(crate) type Retained<'a> = (&'a StorageRecord, Access, BTreeSet<&'a Retainer>);

/// The quotas' own state: every GPU's total storage, the limits the epochs
/// state and the processes' reservations. The storages the rest is derived
/// from are the server's.
#[derive(Debug, Default, Clone, Serialize, Deserialize)]
pub(crate) struct Quotas {
    /// Each GPU's total storage, as the last process that reserved on it
    /// stated it.
    totals: BTreeMap<[u8; 16], u64>,
    /// The limit each epoch of a deployment states for a GPU, by
    /// `(deployment, GPU, epoch)`.
    limits: BTreeMap<(String, [u8; 16], u64), u64>,
    /// The reservations of each connected process, with what it is.
    reservations: BTreeMap<ClientId, (Reserver, ItemBytes)>,
    /// The reservations of each scope and epoch whose worker's process was
    /// lost, per GPU and item, until a process of the scope and epoch
    /// reserves the item again or the epoch retires.
    #[serde(default)]
    lost: BTreeMap<(Scope, u64), ItemBytes>,
}

/// A reservation's identity within a quota, under which its bytes add up.
type ReservationKey = (Item, Option<Scope>, Vec<u64>, Option<Segment>);

impl Quotas {
    /// Applies `client`'s reservations on `gpu` and returns the state
    /// before them, for the caller to restore when it refuses the result.
    pub(crate) fn reserve(
        &mut self,
        client: ClientId,
        reserver: Reserver,
        gpu: [u8; 16],
        total: u64,
        limit: Option<u64>,
        items: &[(Item, u64)],
    ) -> Self {
        let before = self.clone();

        if total > 0 {
            self.totals.insert(gpu, total);
        }
        if let (Reserver::Worker { scope, epoch }, Some(limit)) = (&reserver, limit) {
            self.limits
                .insert((scope.deployment.clone(), gpu, *epoch), limit);
        }

        // A process of a scope and epoch takes each item it reserves over
        // from the reservation its lost predecessor left.
        if let Reserver::Worker { scope, epoch } = &reserver
            && let Some(lost) = self.lost.get_mut(&(scope.clone(), *epoch))
        {
            for &(item, _) in items {
                lost.remove(&(gpu, item));
            }
            if lost.is_empty() {
                self.lost.remove(&(scope.clone(), *epoch));
            }
        }

        let (_, reserved) = self
            .reservations
            .entry(client)
            .or_insert_with(|| (reserver, BTreeMap::new()));
        for &(item, bytes) in items {
            if bytes == 0 {
                reserved.remove(&(gpu, item));
            } else {
                reserved.insert((gpu, item), bytes);
            }
        }

        before
    }

    /// The connection of `client` ends: its process is gone. A worker's
    /// remaining reservations that no other process of its scope and epoch
    /// holds (one that ended in order released them first) are kept for the
    /// process that replaces it; the Transfer Server's end.
    pub(crate) fn disconnect(&mut self, client: ClientId) {
        let Some((Reserver::Worker { scope, epoch }, items)) = self.reservations.remove(&client)
        else {
            return;
        };

        let served: BTreeSet<([u8; 16], Item)> = self
            .reservations
            .values()
            .filter(|(reserver, _)| match reserver {
                Reserver::Worker {
                    scope: of,
                    epoch: at,
                } => *of == scope && *at == epoch,
                Reserver::Transfer => false,
            })
            .flat_map(|(_, items)| items.keys().copied())
            .collect();
        let unserved: Vec<_> = items
            .into_iter()
            .filter(|(key, _)| !served.contains(key))
            .collect();
        if unserved.is_empty() {
            return;
        }

        let lost = self.lost.entry((scope, epoch)).or_default();
        for (key, bytes) in unserved {
            let counted = lost.entry(key).or_default();
            *counted = (*counted).max(bytes);
        }
    }

    /// The limits the epochs of `retired` stated, and the reservations kept
    /// for their lost processes, end with their storage.
    pub(crate) fn retire(&mut self, deployment: &str, retired: &BTreeSet<u64>) {
        self.limits
            .retain(|(of, _, epoch), _| of != deployment || !retired.contains(epoch));
        self.lost
            .retain(|(scope, epoch), _| scope.deployment != deployment || !retired.contains(epoch));
    }

    /// Every limit and kept reservation of `deployment` ends: it is no
    /// longer served.
    pub(crate) fn release(&mut self, deployment: &str) {
        self.limits.retain(|(of, _, _), _| of != deployment);
        self.lost
            .retain(|(scope, _), _| scope.deployment != deployment);
    }

    /// The quotas of the GPUs `gpus` admits and the deployments `deployment`
    /// admits, derived from the retained device storages `retained` and the
    /// processes' reservations, sorted by GPU and deployment, each with the
    /// bytes the Transfer Server's CUDA context takes on its GPU while held
    /// (`contexts`, as it measured them). A GPU that only the Transfer
    /// Server uses has a quota of no deployment.
    pub(crate) fn quotas<'a>(
        &self,
        retained: impl IntoIterator<Item = Retained<'a>>,
        contexts: &[([u8; 16], u64)],
        gpus: &dyn Fn(&[u8; 16]) -> bool,
        deployment: &dyn Fn(&str) -> bool,
    ) -> Vec<Quota> {
        let mut usage: BTreeMap<(String, [u8; 16]), BTreeMap<ReservationKey, u64>> =
            BTreeMap::new();

        // Each retained storage once per deployment that retains it.
        for (record, access, retainers) in retained {
            let ::storage::protocol::Location::Device { gpu } = record.location else {
                continue;
            };
            if !gpus(&gpu) {
                continue;
            }
            let item = match (access, record.segment) {
                (Access::ReadOnly, _) => Item::Weights,
                (Access::Exclusive, Some(_)) => Item::CacheSegment,
                (Access::Exclusive, None) => Item::Exclusive,
            };
            let mut by_deployment: BTreeMap<&str, Vec<&Retainer>> = BTreeMap::new();
            for retainer in &retainers {
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
                let epochs: Vec<u64> = retainers
                    .iter()
                    .map(|retainer| retainer.epoch)
                    .collect::<BTreeSet<_>>()
                    .into_iter()
                    .collect();
                let segment = record.segment.filter(|_| item == Item::CacheSegment);
                *usage
                    .entry((name.to_owned(), gpu))
                    .or_default()
                    .entry((item, scope, epochs, segment))
                    .or_default() += record.bytes;
            }
        }

        // Each worker's reservation, the largest of its scope and epoch's
        // processes and of what a lost process of theirs left; the Transfer
        // Server's, the largest of its processes.
        let mut workers: BTreeMap<(String, [u8; 16], ReservationKey), u64> = BTreeMap::new();
        let mut transfer: BTreeMap<([u8; 16], Item), u64> = BTreeMap::new();
        for ((scope, epoch), items) in &self.lost {
            if !deployment(&scope.deployment) {
                continue;
            }
            for (&(gpu, item), &bytes) in items {
                if !gpus(&gpu) {
                    continue;
                }
                let key = (item, Some(scope.clone()), vec![*epoch], None);
                let counted = workers
                    .entry((scope.deployment.clone(), gpu, key))
                    .or_default();
                *counted = (*counted).max(bytes);
            }
        }
        for (reserver, items) in self.reservations.values() {
            for (&(gpu, item), &bytes) in items {
                if !gpus(&gpu) {
                    continue;
                }
                match reserver {
                    Reserver::Worker { scope, epoch } => {
                        if !deployment(&scope.deployment) {
                            continue;
                        }
                        let key = (item, Some(scope.clone()), vec![*epoch], None);
                        let counted = workers
                            .entry((scope.deployment.clone(), gpu, key))
                            .or_default();
                        *counted = (*counted).max(bytes);
                    }
                    Reserver::Transfer => {
                        let counted = transfer.entry((gpu, item)).or_default();
                        *counted = (*counted).max(bytes);
                    }
                }
            }
        }
        for ((name, gpu, key), bytes) in workers {
            *usage
                .entry((name, gpu))
                .or_default()
                .entry(key)
                .or_default() += bytes;
        }

        // A deployment that stated a limit for a GPU has a quota there even
        // before it holds anything.
        for (name, gpu, _) in self.limits.keys() {
            if gpus(gpu) && deployment(name) {
                usage.entry((name.clone(), *gpu)).or_default();
            }
        }

        // The Transfer Server's items count against every quota of their
        // GPU; a GPU no deployment uses has a quota of its own for them.
        let transfer_gpus: BTreeSet<[u8; 16]> = transfer.keys().map(|(gpu, _)| *gpu).collect();
        for gpu in transfer_gpus {
            let mut names: Vec<String> = usage
                .keys()
                .filter(|(_, of)| *of == gpu)
                .map(|(name, _)| name.clone())
                .collect();
            if names.is_empty() && deployment("") {
                names.push(String::new());
            }
            for name in names {
                let reservations = usage.entry((name, gpu)).or_default();
                for (&(of, item), &bytes) in &transfer {
                    if of == gpu {
                        *reservations
                            .entry((item, None, Vec::new(), None))
                            .or_default() += bytes;
                    }
                }
            }
        }

        usage
            .into_iter()
            .map(|((name, gpu), reservations)| Quota {
                gpu,
                limit: self
                    .limits
                    .iter()
                    .filter(|((of, at, _), _)| *of == name && *at == gpu)
                    .map(|(_, limit)| *limit)
                    .min(),
                deployment: name,
                transfer_context: contexts
                    .iter()
                    .find(|(of, _)| *of == gpu)
                    .map_or(0, |&(_, bytes)| bytes),
                total: self.totals.get(&gpu).copied().unwrap_or(0),
                // The server sets it from the Transfer Server's links.
                host_link_bytes_per_second: None,
                reservations: reservations
                    .into_iter()
                    .map(|((item, scope, epochs, segment), bytes)| Reservation {
                        item,
                        scope,
                        epochs,
                        segment,
                        bytes,
                    })
                    .collect(),
            })
            .collect()
    }
}

/// The first quota of `after` whose usage a change takes past its limit,
/// beyond its usage in `before`: a quota already past its limit (as one
/// whose epochs stated a smaller fraction since) may still shrink, but not
/// grow. Returns the refusal's message, which names every reservation of
/// the quota.
pub(crate) fn overrun(before: &[Quota], after: &[Quota]) -> Option<String> {
    after.iter().find_map(|quota| {
        let limit = quota.limit?;
        let usage = quota.usage();
        let previous = before
            .iter()
            .find(|earlier| earlier.gpu == quota.gpu && earlier.deployment == quota.deployment)
            .map_or(0, Quota::usage);
        (usage > limit && usage > previous).then(|| {
            format!(
                "the reservations on GPU {} would take {usage} bytes, past its quota of \
                 {limit} bytes by {}:\n{}",
                ::storage::protocol::gpu_name(&quota.gpu),
                usage - limit,
                quota.describe()
            )
        })
    })
}
